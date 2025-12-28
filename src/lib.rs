// Hide badges from generated docs
//! <style> .badges { display: none; } </style>

#![no_std]
#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://www.rust-lang.org/logos/rust-logo-128x128-blk.png",
    html_favicon_url = "https://www.rust-lang.org/favicon.ico"
)]
#![deny(
    missing_docs,
    missing_debug_implementations,
    clippy::undocumented_unsafe_blocks
)]

use core::{convert::Infallible, fmt, ops::DerefMut};

pub mod block;
pub mod utils;
mod word;

/// A potentially fallible variant of [`RngCore`]
///
/// This trait is a generalization of [`RngCore`] to support potentially-
/// fallible IO-based generators such as [`OsRng`].
///
/// All implementations of [`RngCore`] automatically support this `TryRngCore`
/// trait, using [`Infallible`][core::convert::Infallible] as the associated
/// `Error` type.
///
/// An implementation of this trait may be made compatible with code requiring
/// an [`RngCore`] through [`TryRngCore::unwrap_err`]. The resulting RNG will
/// panic in case the underlying fallible RNG yields an error.
///
/// [`OsRng`]: https://docs.rs/rand/latest/rand/rngs/struct.OsRng.html
pub trait TryRngCore {
    /// The type returned in the event of a RNG error.
    type Error: fmt::Debug + fmt::Display;

    /// Return the next random `u32`.
    fn try_next_u32(&mut self) -> Result<u32, Self::Error>;
    /// Return the next random `u64`.
    fn try_next_u64(&mut self) -> Result<u64, Self::Error>;
    /// Fill `dest` entirely with random data.
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error>;

    /// Wrap RNG with the [`UnwrapErr`] wrapper.
    fn unwrap_err(self) -> UnwrapErr<Self>
    where
        Self: Sized,
    {
        UnwrapErr(self)
    }
}

impl<T: DerefMut> TryRngCore for T
where
    T::Target: TryRngCore,
{
    type Error = <T::Target as TryRngCore>::Error;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        self.deref_mut().try_next_u32()
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.deref_mut().try_next_u64()
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        self.deref_mut().try_fill_bytes(dst)
    }
}

/// An infallible [`TryRngCore`]
pub trait RngCore: TryRngCore<Error = Infallible> {}
impl<R: TryRngCore<Error = Infallible>> RngCore for R {}

/// A marker trait over [`TryRngCore`] for securely unpredictable RNGs
///
/// This trait is like [`CryptoRng`] but for the trait [`TryRngCore`].
///
/// This marker trait indicates that the implementing generator is intended,
/// when correctly seeded and protected from side-channel attacks such as a
/// leaking of state, to be a cryptographically secure generator. This trait is
/// provided as a tool to aid review of cryptographic code, but does not by
/// itself guarantee suitability for cryptographic applications.
///
/// Implementors of `TryCryptoRng` should only implement [`Default`] if the
/// `default()` instances are themselves secure generators: for example if the
/// implementing type is a stateless interface over a secure external generator
/// (like [`OsRng`]) or if the `default()` instance uses a strong, fresh seed.
///
/// [`OsRng`]: https://docs.rs/rand/latest/rand/rngs/struct.OsRng.html
pub trait TryCryptoRng: TryRngCore {}

/// Wrapper around [`TryRngCore`] implementation which implements [`RngCore`]
/// by panicking on potential errors.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Hash)]
pub struct UnwrapErr<R: TryRngCore>(pub R);

impl<R: TryRngCore> TryRngCore for UnwrapErr<R> {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.0.try_next_u32().unwrap())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.0.try_next_u64().unwrap())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        Ok(self.0.try_fill_bytes(dst).unwrap())
    }
}

impl<R: TryCryptoRng> TryCryptoRng for UnwrapErr<R> {}

impl<'r, R: TryRngCore + ?Sized> UnwrapErr<&'r mut R> {
    /// Reborrow with a new lifetime
    ///
    /// Rust allows references like `&T` or `&mut T` to be "reborrowed" through
    /// coercion: essentially, the pointer is copied under a new, shorter, lifetime.
    /// Until rfcs#1403 lands, reborrows on user types require a method call.
    #[inline(always)]
    pub fn re<'b>(&'b mut self) -> UnwrapErr<&'b mut R>
    where
        'r: 'b,
    {
        UnwrapErr(&mut self.0)
    }
}

/// A random number generator that can be explicitly seeded.
///
/// This trait encapsulates the low-level functionality common to all
/// pseudo-random number generators (PRNGs, or algorithmic generators).
///
/// A generator implementing `SeedableRng` will usually be deterministic, but
/// beware that portability and reproducibility of results **is not implied**.
/// Refer to documentation of the generator, noting that generators named after
/// a specific algorithm are usually tested for reproducibility against a
/// reference vector, while `SmallRng` and `StdRng` specifically opt out of
/// reproducibility guarantees.
///
/// [`rand`]: https://docs.rs/rand
pub trait SeedableRng: Sized {
    /// Seed type, which is restricted to types mutably-dereferenceable as `u8`
    /// arrays (we recommend `[u8; N]` for some `N`).
    ///
    /// It is recommended to seed PRNGs with a seed of at least circa 100 bits,
    /// which means an array of `[u8; 12]` or greater to avoid picking RNGs with
    /// partially overlapping periods.
    ///
    /// For cryptographic RNG's a seed of 256 bits is recommended, `[u8; 32]`.
    ///
    ///
    /// # Implementing `SeedableRng` for RNGs with large seeds
    ///
    /// Note that [`Default`] is not implemented for large arrays `[u8; N]` with
    /// `N` > 32. To be able to implement the traits required by `SeedableRng`
    /// for RNGs with such large seeds, the newtype pattern can be used:
    ///
    /// ```
    /// use rand_core::SeedableRng;
    ///
    /// const N: usize = 64;
    /// #[derive(Clone)]
    /// pub struct MyRngSeed(pub [u8; N]);
    /// # #[allow(dead_code)]
    /// pub struct MyRng(MyRngSeed);
    ///
    /// impl Default for MyRngSeed {
    ///     fn default() -> MyRngSeed {
    ///         MyRngSeed([0; N])
    ///     }
    /// }
    ///
    /// impl AsRef<[u8]> for MyRngSeed {
    ///     fn as_ref(&self) -> &[u8] {
    ///         &self.0
    ///     }
    /// }
    ///
    /// impl AsMut<[u8]> for MyRngSeed {
    ///     fn as_mut(&mut self) -> &mut [u8] {
    ///         &mut self.0
    ///     }
    /// }
    ///
    /// impl SeedableRng for MyRng {
    ///     type Seed = MyRngSeed;
    ///
    ///     fn from_seed(seed: MyRngSeed) -> MyRng {
    ///         MyRng(seed)
    ///     }
    /// }
    /// ```
    type Seed: Clone + Default + AsRef<[u8]> + AsMut<[u8]>;

    /// Create a new PRNG using the given seed.
    ///
    /// PRNG implementations are allowed to assume that bits in the seed are
    /// well distributed. That means usually that the number of one and zero
    /// bits are roughly equal, and values like 0, 1 and (size - 1) are unlikely.
    /// Note that many non-cryptographic PRNGs will show poor quality output
    /// if this is not adhered to. If you wish to seed from simple numbers, use
    /// `seed_from_u64` instead.
    ///
    /// All PRNG implementations should be reproducible unless otherwise noted:
    /// given a fixed `seed`, the same sequence of output should be produced
    /// on all runs, library versions and architectures (e.g. check endianness).
    /// Any "value-breaking" changes to the generator should require bumping at
    /// least the minor version and documentation of the change.
    ///
    /// It is not required that this function yield the same state as a
    /// reference implementation of the PRNG given equivalent seed; if necessary
    /// another constructor replicating behaviour from a reference
    /// implementation can be added.
    ///
    /// PRNG implementations should make sure `from_seed` never panics. In the
    /// case that some special values (like an all zero seed) are not viable
    /// seeds it is preferable to map these to alternative constant value(s),
    /// for example `0xBAD5EEDu32` or `0x0DDB1A5E5BAD5EEDu64` ("odd biases? bad
    /// seed"). This is assuming only a small number of values must be rejected.
    fn from_seed(seed: Self::Seed) -> Self;

    /// Create a new PRNG using a `u64` seed.
    ///
    /// This is a convenience-wrapper around `from_seed` to allow construction
    /// of any `SeedableRng` from a simple `u64` value. It is designed such that
    /// low Hamming Weight numbers like 0 and 1 can be used and should still
    /// result in good, independent seeds to the PRNG which is returned.
    ///
    /// This **is not suitable for cryptography**, as should be clear given that
    /// the input size is only 64 bits.
    ///
    /// Implementations for PRNGs *may* provide their own implementations of
    /// this function, but the default implementation should be good enough for
    /// all purposes. *Changing* the implementation of this function should be
    /// considered a value-breaking change.
    fn seed_from_u64(mut state: u64) -> Self {
        // We use PCG32 to generate a u32 sequence, and copy to the seed
        fn pcg32(state: &mut u64) -> [u8; 4] {
            const MUL: u64 = 6364136223846793005;
            const INC: u64 = 11634580027462260723;

            // We advance the state first (to get away from the input value,
            // in case it has low Hamming Weight).
            *state = state.wrapping_mul(MUL).wrapping_add(INC);
            let state = *state;

            // Use PCG output function with to_le to generate x:
            let xorshifted = (((state >> 18) ^ state) >> 27) as u32;
            let rot = (state >> 59) as u32;
            let x = xorshifted.rotate_right(rot);
            x.to_le_bytes()
        }

        let mut seed = Self::Seed::default();
        let mut iter = seed.as_mut().chunks_exact_mut(4);
        for chunk in &mut iter {
            chunk.copy_from_slice(&pcg32(&mut state));
        }
        let rem = iter.into_remainder();
        if !rem.is_empty() {
            rem.copy_from_slice(&pcg32(&mut state)[..rem.len()]);
        }

        Self::from_seed(seed)
    }

    /// Create a new PRNG seeded from an infallible `Rng`.
    ///
    /// This may be useful when needing to rapidly seed many PRNGs from a master
    /// PRNG, and to allow forking of PRNGs. It may be considered deterministic.
    ///
    /// The master PRNG should be at least as high quality as the child PRNGs.
    /// When seeding non-cryptographic child PRNGs, we recommend using a
    /// different algorithm for the master PRNG (ideally a CSPRNG) to avoid
    /// correlations between the child PRNGs. If this is not possible (e.g.
    /// forking using small non-crypto PRNGs) ensure that your PRNG has a good
    /// mixing function on the output or consider use of a hash function with
    /// `from_seed`.
    ///
    /// Note that seeding `XorShiftRng` from another `XorShiftRng` provides an
    /// extreme example of what can go wrong: the new PRNG will be a clone
    /// of the parent.
    ///
    /// PRNG implementations are allowed to assume that a good RNG is provided
    /// for seeding, and that it is cryptographically secure when appropriate.
    /// As of `rand` 0.7 / `rand_core` 0.5, implementations overriding this
    /// method should ensure the implementation satisfies reproducibility
    /// (in prior versions this was not required).
    ///
    /// [`rand`]: https://docs.rs/rand
    fn from_rng<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let mut seed = Self::Seed::default();
        match rng.try_fill_bytes(seed.as_mut()) {
            Ok(()) => {}
        }
        Self::from_seed(seed)
    }

    /// Create a new PRNG seeded from a potentially fallible `Rng`.
    ///
    /// See [`from_rng`][SeedableRng::from_rng] docs for more information.
    fn try_from_rng<R: TryRngCore + ?Sized>(rng: &mut R) -> Result<Self, R::Error> {
        let mut seed = Self::Seed::default();
        rng.try_fill_bytes(seed.as_mut())?;
        Ok(Self::from_seed(seed))
    }

    /// Fork this PRNG
    ///
    /// This creates a new PRNG from the current one by initializing a new one and
    /// seeding it from the current one.
    ///
    /// This is useful when initializing a PRNG for a thread
    fn fork(&mut self) -> Self
    where
        Self: RngCore,
    {
        Self::from_rng(self)
    }

    /// Fork this PRNG
    ///
    /// This creates a new PRNG from the current one by initializing a new one and
    /// seeding it from the current one.
    ///
    /// This is useful when initializing a PRNG for a thread.
    ///
    /// This is the failable equivalent to [`SeedableRng::fork`]
    fn try_fork(&mut self) -> Result<Self, Self::Error>
    where
        Self: TryRngCore,
    {
        Self::try_from_rng(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_seed_from_u64() {
        struct SeedableNum(u64);
        impl SeedableRng for SeedableNum {
            type Seed = [u8; 8];

            fn from_seed(seed: Self::Seed) -> Self {
                let x: [u64; 1] = utils::read_words(&seed);
                SeedableNum(x[0])
            }
        }

        const N: usize = 8;
        const SEEDS: [u64; N] = [0u64, 1, 2, 3, 4, 8, 16, -1i64 as u64];
        let mut results = [0u64; N];
        for (i, seed) in SEEDS.iter().enumerate() {
            let SeedableNum(x) = SeedableNum::seed_from_u64(*seed);
            results[i] = x;
        }

        for (i1, r1) in results.iter().enumerate() {
            let weight = r1.count_ones();
            // This is the binomial distribution B(64, 0.5), so chance of
            // weight < 20 is binocdf(19, 64, 0.5) = 7.8e-4, and same for
            // weight > 44.
            assert!((20..=44).contains(&weight));

            for (i2, r2) in results.iter().enumerate() {
                if i1 == i2 {
                    continue;
                }
                let diff_weight = (r1 ^ r2).count_ones();
                assert!(diff_weight >= 20);
            }
        }

        // value-breakage test:
        assert_eq!(results[0], 5029875928683246316);
    }

    // A stub RNG.
    struct SomeRng;

    impl TryRngCore for SomeRng {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
            unimplemented!()
        }
        fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
            unimplemented!()
        }
        fn try_fill_bytes(&mut self, _dst: &mut [u8]) -> Result<(), Self::Error> {
            unimplemented!()
        }
    }

    impl TryCryptoRng for SomeRng {}

    #[test]
    fn dyn_rngcore_to_tryrngcore() {
        // Illustrates the need for `+ ?Sized` bound in `impl<R: RngCore> TryRngCore for R`.

        // A method in another crate taking a fallible RNG
        fn third_party_api(_rng: &mut (impl TryRngCore + ?Sized)) -> bool {
            true
        }

        // A method in our crate requiring an infallible RNG
        fn my_api(rng: &mut dyn RngCore) -> bool {
            // We want to call the method above
            third_party_api(rng)
        }

        assert!(my_api(&mut SomeRng));
    }

    #[test]
    fn dyn_unwrap_mut_tryrngcore() {
        // Illustrates the need for `+ ?Sized` bound in
        // `impl<R: TryRngCore> RngCore for UnwrapMut<'_, R>`.

        fn third_party_api(_rng: &mut impl RngCore) -> bool {
            true
        }

        fn my_api(rng: &mut (impl TryRngCore + ?Sized)) -> bool {
            let mut infallible_rng = rng.unwrap_err();
            third_party_api(&mut infallible_rng)
        }

        assert!(my_api(&mut SomeRng));
    }

    #[test]
    fn reborrow_unwrap_mut() {
        struct FourRng;

        impl TryRngCore for FourRng {
            type Error = core::convert::Infallible;
            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                Ok(4)
            }
            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                unimplemented!()
            }
            fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Self::Error> {
                unimplemented!()
            }
        }

        let mut rng = FourRng;
        let mut rng = (&mut rng).unwrap_err();

        assert_eq!(rng.try_next_u32(), Ok(4));
        {
            let mut rng2 = rng.re();
            assert_eq!(rng2.try_next_u32(), Ok(4));
            // Make sure rng2 is dropped.
        }
        assert_eq!(rng.try_next_u32(), Ok(4));
    }
}
