from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time


def get_prime(size):
    """Génère un nombre premier de taille donnée"""
    seed = random_state(int(time.time() * 1000000))
    p = mpz_urandomb(seed, size)
    p = p.bit_set(size - 1)  # Set MSB to 1 
    return next_prime(p)

def generate_keys(size) -> dict:
    """Génère les clés publiques et privées de Paillier"""
    p = get_prime(size//2)
    while True:
        q = get_prime(size//2)
        N = p * q
        phi = (p-1) * (q-1)
        if gcd(N, phi) == 1 and p != q:
            break
    g = 1 + N
    pub_key = (N, g)
    priv_key = (phi, max(p,q), min(p,q))
    return {"public": pub_key, "secret": priv_key}

def get_r(N):
    """Génère un nombre aléatoire r copremier avec N"""
    while True:
        seed = random_state(int(time.time() * 1000000))
        r = mpz_random(seed, N)
        if gcd(r, N) == 1:
            break
    return r

def encrypt(message, pub_key):
    """Chiffre un message avec Paillier"""
    N, g = pub_key
    r = get_r(N)
    N2 = N ** 2
    message = mpz(message)
    
    # c = g^m * r^N mod N^2
    g_pow_m = powmod(g, message, N2)
    r_pow_N = powmod(r, N, N2)
    c = (g_pow_m * r_pow_N) % N2
    return c

def encrypt_given_r(message, public_key, r):
    """Chiffre un message avec Paillier pour une valeur de r donnée"""
    N, g = public_key
    N2 = N ** 2
    message = mpz(message)
    
    g_pow_m = powmod(g, message, N2)
    r_pow_N = powmod(r, N, N2)
    c = (g_pow_m * r_pow_N) % N2
    return c

def decrypt_CRT(enc, priv_key, pub_key):
    """Déchiffre avec CRT (plus rapide)"""
    phi, p, q = priv_key
    N = pub_key[0]
    
    ## Calcul avec CRT
    xp = powmod(enc, phi, p**2)
    xq = powmod(enc, phi, q**2)
    
    # Inverse de q^2 modulo p^2
    Invq = invert(q**2, p**2)
    
    # Reconstruction CRT
    x = ((Invq*(xp-xq)) % p**2)*q**2 + xq
    
    # Fonction L et déchiffrement final
    L_result = (x-1)//N
    m = (L_result * invert(phi, N)) % N
    return int(m)