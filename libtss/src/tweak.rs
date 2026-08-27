//! Scalar tweaks on DKLs23 key shares.
//!
//! BIP-32 [`derive_child`](crate::derive_child) applies an HMAC-derived
//! additive tweak. Lightning (LND) instead derives per-commitment keys from
//! caller-supplied scalars:
//!
//! - single tweak (additive):        `k' = k + t (mod n)`
//! - revocation / double tweak:      `k' = k * a (mod n)`, then an additive step
//!
//! [`tweak_add`] and [`tweak_mul`] return a new key share whose secret share,
//! public key, BIP-32 derivation data and address are moved by the given map,
//! together with a public key package whose group key and every verifying
//! share are moved by the same map. Because Lagrange coefficients sum to one,
//! `Σ lᵢ·(xᵢ + t) = sk + t` and `Σ lᵢ·(a·xᵢ) = a·sk`, so a threshold set of
//! tweaked shares signs under the tweaked group key. The DKLs23 base-OT
//! correlations, zero-share seeds and multiplication state never depend on the
//! share value — the same property `derive_child` relies on — so they are
//! reused unchanged. The chain code is deliberately left untouched: signing
//! salts the two-party multiplication transcript with it and every party must
//! keep agreeing on it. Every party (and the package) must apply the same
//! tweak before signing; a mismatch aborts recoverably in signing phase 3.
//!
//! For BIP-341 Taproot tweaking of FROST shares see
//! [`frost_tweak_key_share`](crate::frost::frost_tweak_key_share).

use std::collections::BTreeMap;

use dkls23_secp256k1::curve::DklsCurve;
use dkls23_secp256k1::protocols::{Parameters, Party, PartyIndex, PublicKeyPackage};
use k256::elliptic_curve::ops::Reduce;
use k256::elliptic_curve::{CurveGroup, Field, FieldBytes, Group};

use crate::keyshare::{self, KeyShareHandle};
use crate::types::Ciphersuite;
use crate::TssError;

/// Return a key share whose secret is `secret + tweak (mod n)`.
///
/// `tweak` is a 32-byte big-endian scalar, reduced mod n (matching btcec's
/// `ModNScalar` and therefore LND's single-tweak convention). A zero tweak is
/// a no-op. Supported for `Secp256k1ECDSA` and `Secp256r1ECDSA` (DKLs23);
/// other ciphersuites return [`TssError::ProtocolMismatch`]. A tweak whose
/// result is the identity point returns [`TssError::TweakError`]; a scalar of
/// the wrong length returns [`TssError::InvalidConfig`]. The original key
/// share remains valid.
pub fn tweak_add(key_share: &KeyShareHandle, tweak: &[u8]) -> Result<KeyShareHandle, TssError> {
    match key_share.ciphersuite() {
        Ciphersuite::Secp256k1ECDSA => tweak_k1(
            key_share,
            &Tweak::Add(scalar_from_bytes::<k256::Secp256k1>(tweak)?),
        ),
        Ciphersuite::Secp256r1ECDSA => tweak_r1(
            key_share,
            &Tweak::Add(scalar_from_bytes::<p256::NistP256>(tweak)?),
        ),
        _ => Err(TssError::ProtocolMismatch),
    }
}

/// Return a key share whose secret is `secret * factor (mod n)`.
///
/// `factor` is a 32-byte big-endian scalar, reduced mod n. It is the building
/// block of LND's revocation ("double") tweak. A factor that reduces to zero
/// would destroy the key and returns [`TssError::TweakError`]; a factor of one
/// is a no-op. Ciphersuite support and error mapping match [`tweak_add`].
pub fn tweak_mul(key_share: &KeyShareHandle, factor: &[u8]) -> Result<KeyShareHandle, TssError> {
    match key_share.ciphersuite() {
        Ciphersuite::Secp256k1ECDSA => tweak_k1(
            key_share,
            &Tweak::Mul(scalar_from_bytes::<k256::Secp256k1>(factor)?),
        ),
        Ciphersuite::Secp256r1ECDSA => tweak_r1(
            key_share,
            &Tweak::Mul(scalar_from_bytes::<p256::NistP256>(factor)?),
        ),
        _ => Err(TssError::ProtocolMismatch),
    }
}

enum Tweak<C: DklsCurve> {
    Add(C::Scalar),
    Mul(C::Scalar),
}

impl<C: DklsCurve> Tweak<C> {
    fn validate(&self) -> Result<(), TssError> {
        match self {
            Tweak::Mul(factor) if bool::from(factor.is_zero()) => Err(TssError::TweakError(
                "multiplicative factor reduces to zero".into(),
            )),
            _ => Ok(()),
        }
    }

    fn apply_scalar(&self, value: C::Scalar) -> C::Scalar {
        match self {
            Tweak::Add(tweak) => value + *tweak,
            Tweak::Mul(factor) => value * *factor,
        }
    }

    fn apply_point(&self, point: C::AffinePoint) -> C::ProjectivePoint {
        let point = C::ProjectivePoint::from(point);
        match self {
            Tweak::Add(tweak) => point + C::ProjectivePoint::generator() * *tweak,
            Tweak::Mul(factor) => point * *factor,
        }
    }
}

fn tweak_k1(
    key_share: &KeyShareHandle,
    tweak: &Tweak<k256::Secp256k1>,
) -> Result<KeyShareHandle, TssError> {
    let party = keyshare::clone_dkls_party(key_share)?;
    let pubkey = keyshare::clone_dkls_pubkey(key_share)?;
    let (party, pubkey) =
        apply_tweak(party, &pubkey, tweak, dkls23_secp256k1::compute_eth_address)?;
    keyshare::from_dkls_dkg(party, pubkey)
}

fn tweak_r1(
    key_share: &KeyShareHandle,
    tweak: &Tweak<p256::NistP256>,
) -> Result<KeyShareHandle, TssError> {
    let party = keyshare::clone_dkls_r1_party(key_share)?;
    let pubkey = keyshare::clone_dkls_r1_pubkey(key_share)?;
    let (party, pubkey) = apply_tweak(
        party,
        &pubkey,
        tweak,
        dkls23_secp256r1::compute_neo3_address,
    )?;
    keyshare::from_dkls_r1_dkg(party, pubkey)
}

fn scalar_from_bytes<C: DklsCurve>(bytes: &[u8]) -> Result<C::Scalar, TssError> {
    let field_bytes: &FieldBytes<C> = bytes.try_into().map_err(|_| {
        TssError::InvalidConfig(format!(
            "tweak must be {} bytes, got {}",
            FieldBytes::<C>::default().len(),
            bytes.len()
        ))
    })?;
    Ok(<C::Scalar as Reduce<FieldBytes<C>>>::reduce(field_bytes))
}

fn apply_tweak<C: DklsCurve>(
    mut party: Party<C>,
    pubkey: &PublicKeyPackage<C>,
    tweak: &Tweak<C>,
    address_fn: impl Fn(&C::AffinePoint) -> String,
) -> Result<(Party<C>, PublicKeyPackage<C>), TssError> {
    tweak.validate()?;

    let tweaked_pk = tweak.apply_point(party.pk);
    if bool::from(tweaked_pk.is_identity()) {
        return Err(TssError::TweakError(
            "tweak produces the identity public key".into(),
        ));
    }
    let tweaked_pk = tweaked_pk.to_affine();
    let tweaked_poly_point = tweak.apply_scalar(party.poly_point);

    party.poly_point = tweaked_poly_point;
    party.pk = tweaked_pk;
    party.derivation_data.poly_point = tweaked_poly_point;
    party.derivation_data.pk = tweaked_pk;
    party.address = address_fn(&tweaked_pk);

    let verifying_key = tweak.apply_point(*pubkey.verifying_key()).to_affine();
    let mut verifying_shares = BTreeMap::new();
    for index in 1..=pubkey.share_count() {
        let party_index = PartyIndex::new(index).map_err(|_| TssError::InvalidIdentifier)?;
        let share = pubkey
            .verifying_share(party_index)
            .ok_or(TssError::InvalidShare)?;
        verifying_shares.insert(party_index, tweak.apply_point(*share).to_affine());
    }
    let parameters = Parameters::new(pubkey.threshold(), pubkey.share_count())
        .map_err(|err| TssError::InvalidConfig(err.to_string()))?;

    Ok((
        party,
        PublicKeyPackage::new(verifying_key, verifying_shares, parameters),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::{ethereum_address, neo3_address};
    use crate::Identifier;
    use dkls23_secp256k1::protocols::re_key::re_key;
    use k256::elliptic_curve::sec1::ToSec1Point;
    use k256::{ProjectivePoint, PublicKey, Scalar};

    const SECRET: u64 = 15;
    const SHARE_COUNT: u16 = 3;

    fn dkls_handle() -> KeyShareHandle {
        let parameters = Parameters::new(2, SHARE_COUNT as u8).unwrap();
        let (parties, pubkeys) = re_key(
            &parameters,
            b"session",
            &Scalar::from(SECRET),
            Some([3u8; 32]),
            dkls23_secp256k1::compute_eth_address,
        );

        keyshare::from_dkls_dkg(parties[0].clone(), pubkeys).unwrap()
    }

    fn dkls_r1_handle() -> KeyShareHandle {
        let parameters = Parameters::new(2, SHARE_COUNT as u8).unwrap();
        let (parties, pubkeys) = dkls23_secp256r1::protocols::re_key::re_key(
            &parameters,
            b"session",
            &p256::Scalar::from(SECRET),
            Some([3u8; 32]),
            dkls23_secp256r1::compute_neo3_address,
        );

        keyshare::from_dkls_r1_dkg(parties[0].clone(), pubkeys).unwrap()
    }

    fn scalar_bytes(value: u64) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[24..].copy_from_slice(&value.to_be_bytes());
        bytes
    }

    fn k1_point(compressed: &[u8]) -> ProjectivePoint {
        PublicKey::from_sec1_bytes(compressed)
            .unwrap()
            .to_projective()
    }

    fn compressed(point: ProjectivePoint) -> Vec<u8> {
        point.to_affine().to_sec1_point(true).as_bytes().to_vec()
    }

    #[test]
    fn tweak_add_k1_shifts_group_key_and_every_verifying_share() {
        let handle = dkls_handle();
        let tweaked = tweak_add(&handle, &scalar_bytes(7)).unwrap();
        let shift = ProjectivePoint::GENERATOR * Scalar::from(7u64);

        assert_eq!(tweaked.ciphersuite(), Ciphersuite::Secp256k1ECDSA);
        assert_eq!(tweaked.identifier(), handle.identifier());
        assert_eq!(
            tweaked.group_verifying_key(),
            compressed(k1_point(&handle.group_verifying_key()) + shift)
        );
        assert_eq!(
            tweaked.verifying_share(),
            compressed(k1_point(&handle.verifying_share()) + shift)
        );
        for index in 1..=SHARE_COUNT {
            let id = Identifier::new(index).unwrap();
            let before = handle.public_key_package().verifying_share(id).unwrap();
            let after = tweaked.public_key_package().verifying_share(id).unwrap();
            assert_eq!(after.to_vec(), compressed(k1_point(before) + shift));
        }
    }

    #[test]
    fn tweak_mul_k1_scales_group_key_and_every_verifying_share() {
        let handle = dkls_handle();
        let tweaked = tweak_mul(&handle, &scalar_bytes(3)).unwrap();
        let factor = Scalar::from(3u64);

        assert_eq!(
            tweaked.group_verifying_key(),
            compressed(k1_point(&handle.group_verifying_key()) * factor)
        );
        for index in 1..=SHARE_COUNT {
            let id = Identifier::new(index).unwrap();
            let before = handle.public_key_package().verifying_share(id).unwrap();
            let after = tweaked.public_key_package().verifying_share(id).unwrap();
            assert_eq!(after.to_vec(), compressed(k1_point(before) * factor));
        }
    }

    #[test]
    fn tweak_add_updates_ethereum_address() {
        let handle = dkls_handle();
        let tweaked = tweak_add(&handle, &scalar_bytes(7)).unwrap();
        let party = keyshare::clone_dkls_party(&tweaked).unwrap();

        assert_ne!(
            ethereum_address(&tweaked).unwrap(),
            ethereum_address(&handle).unwrap()
        );
        assert_eq!(
            ethereum_address(&tweaked).unwrap(),
            dkls23_secp256k1::compute_eth_address(&party.pk)
        );
        assert_eq!(
            party.pk.to_sec1_point(true).as_bytes(),
            tweaked.group_verifying_key().as_slice()
        );
    }

    #[test]
    fn tweak_preserves_derivation_metadata_and_session() {
        let handle = dkls_handle();
        let before = keyshare::clone_dkls_party(&handle).unwrap();
        let tweaked = tweak_mul(&handle, &scalar_bytes(3)).unwrap();
        let after = keyshare::clone_dkls_party(&tweaked).unwrap();

        assert_eq!(
            after.derivation_data.chain_code,
            before.derivation_data.chain_code
        );
        assert_eq!(after.derivation_data.depth, before.derivation_data.depth);
        assert_eq!(
            after.derivation_data.child_number,
            before.derivation_data.child_number
        );
        assert_eq!(
            after.derivation_data.parent_fingerprint,
            before.derivation_data.parent_fingerprint
        );
        assert_eq!(after.session_id, before.session_id);
        assert_eq!(after.party_index, before.party_index);
        assert_eq!(after.derivation_data.poly_point, after.poly_point);
        assert_eq!(after.derivation_data.pk, after.pk);
        assert_ne!(after.poly_point, before.poly_point);
    }

    #[test]
    fn tweak_add_r1_matches_expected_public_key() {
        use p256::elliptic_curve::sec1::ToSec1Point;

        let handle = dkls_r1_handle();
        let tweaked = tweak_add(&handle, &scalar_bytes(7)).unwrap();
        let expected = (p256::PublicKey::from_sec1_bytes(&handle.group_verifying_key())
            .unwrap()
            .to_projective()
            + p256::ProjectivePoint::GENERATOR * p256::Scalar::from(7u64))
        .to_affine();

        assert_eq!(tweaked.ciphersuite(), Ciphersuite::Secp256r1ECDSA);
        assert_eq!(
            tweaked.group_verifying_key(),
            expected.to_sec1_point(true).as_bytes()
        );
        assert_eq!(
            neo3_address(&tweaked).unwrap(),
            dkls23_secp256r1::compute_neo3_address(&expected)
        );
    }

    #[test]
    fn tweak_mul_r1_matches_expected_public_key() {
        use p256::elliptic_curve::sec1::ToSec1Point;

        let handle = dkls_r1_handle();
        let tweaked = tweak_mul(&handle, &scalar_bytes(3)).unwrap();
        let expected = (p256::PublicKey::from_sec1_bytes(&handle.group_verifying_key())
            .unwrap()
            .to_projective()
            * p256::Scalar::from(3u64))
        .to_affine();

        assert_eq!(
            tweaked.group_verifying_key(),
            expected.to_sec1_point(true).as_bytes()
        );
    }

    #[test]
    fn tweak_add_negated_scalar_round_trips() {
        let handle = dkls_handle();
        let tweaked = tweak_add(&handle, &scalar_bytes(7)).unwrap();
        let restored = tweak_add(&tweaked, &(-Scalar::from(7u64)).to_bytes()).unwrap();

        assert_eq!(restored.group_verifying_key(), handle.group_verifying_key());
        assert_eq!(restored.verifying_share(), handle.verifying_share());
        assert_eq!(
            ethereum_address(&restored).unwrap(),
            ethereum_address(&handle).unwrap()
        );
    }

    #[test]
    fn tweak_composition_matches_single_scalar() {
        let handle = dkls_handle();
        let twice = tweak_add(
            &tweak_add(&handle, &scalar_bytes(7)).unwrap(),
            &scalar_bytes(5),
        )
        .unwrap();
        let once = tweak_add(&handle, &scalar_bytes(12)).unwrap();

        assert_eq!(twice.group_verifying_key(), once.group_verifying_key());
        assert_eq!(twice.verifying_share(), once.verifying_share());
    }

    #[test]
    fn tweak_add_zero_and_mul_one_are_identity() {
        let handle = dkls_handle();

        let unchanged = tweak_add(&handle, &scalar_bytes(0)).unwrap();
        assert_eq!(
            unchanged.group_verifying_key(),
            handle.group_verifying_key()
        );
        assert_eq!(unchanged.verifying_share(), handle.verifying_share());

        let unchanged = tweak_mul(&handle, &scalar_bytes(1)).unwrap();
        assert_eq!(
            unchanged.group_verifying_key(),
            handle.group_verifying_key()
        );
        assert_eq!(unchanged.verifying_share(), handle.verifying_share());
    }

    #[test]
    fn tweak_mul_zero_factor_rejected() {
        let handle = dkls_handle();
        assert!(matches!(
            tweak_mul(&handle, &scalar_bytes(0)),
            Err(TssError::TweakError(_))
        ));
        assert!(matches!(
            tweak_mul(&dkls_r1_handle(), &scalar_bytes(0)),
            Err(TssError::TweakError(_))
        ));
    }

    #[test]
    fn tweak_add_identity_result_rejected() {
        let handle = dkls_handle();
        assert!(matches!(
            tweak_add(&handle, &(-Scalar::from(SECRET)).to_bytes()),
            Err(TssError::TweakError(_))
        ));
    }

    #[test]
    fn tweak_rejects_wrong_length_scalar() {
        let handle = dkls_handle();
        assert!(matches!(
            tweak_add(&handle, &[1u8; 31]),
            Err(TssError::InvalidConfig(_))
        ));
        assert!(matches!(
            tweak_mul(&handle, &[1u8; 33]),
            Err(TssError::InvalidConfig(_))
        ));
        assert!(matches!(
            tweak_add(&dkls_r1_handle(), &[]),
            Err(TssError::InvalidConfig(_))
        ));
    }

    #[test]
    fn tweak_unsupported_suite_rejected() {
        for suite in [Ciphersuite::Secp256k1Taproot, Ciphersuite::Ed25519] {
            let config = crate::types::ThresholdConfig {
                min_signers: 2,
                max_signers: 3,
                suite,
            };
            let (shares, pubkey) = crate::frost::frost_generate_with_dealer(&config).unwrap();
            let handle =
                keyshare::from_frost_dkg(shares.into_iter().next().unwrap(), &pubkey).unwrap();

            assert!(matches!(
                tweak_add(&handle, &scalar_bytes(7)),
                Err(TssError::ProtocolMismatch)
            ));
            assert!(matches!(
                tweak_mul(&handle, &scalar_bytes(3)),
                Err(TssError::ProtocolMismatch)
            ));
        }
    }
}
