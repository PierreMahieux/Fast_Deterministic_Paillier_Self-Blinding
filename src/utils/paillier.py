from gmpy2 import mpz, powmod, invert, gcd, mpz_random, random_state, mpz_urandomb, next_prime
import time
import numpy as np

def get_prime(size):
    seed = random_state(int(time.time() * 1000000))
    p = mpz_urandomb(seed, size)
    p = p.bit_set(size - 1)  # Set MSB to 1 
    return next_prime(p)

def generate_keys(size) -> dict:
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

def generate_r(N):
    while True:
        seed = random_state(time.time_ns())
        r = mpz_random(seed, N)
        if gcd(r, N) == 1:
            break
    return r

def encrypt(message, public_key):
    return encrypt_given_r(message, public_key, generate_r(public_key[0]))

def encrypt_given_r(message, public_key, r):
    N2 = public_key[0] ** 2
    r = powmod(r, public_key[0], N2)
    c = powmod(public_key[1], message, N2)
    c = c*r % N2
    return c

def decrypt_CRT(enc, secret_key, public_key):
    xp = powmod(enc, secret_key[0], secret_key[1]**2)
    xq = powmod(enc, secret_key[0], secret_key[2]**2)
    Invq = invert(secret_key[2]**2, secret_key[1]**2)
    x = ((Invq*(xp-xq))% secret_key[1]**2)*secret_key[2]**2 +  xq
    m = ((x-1)//public_key[0]*invert(secret_key[0], public_key[0])) % public_key[0]
    return m % public_key[0]

def decrypt(enc, priv_key, pub_key):
    N2 = pub_key[0] ** 2
    phiInv = invert(priv_key[0], pub_key[0])
    m = powmod(enc, priv_key[0], N2)
    m = m - 1
    m = m//pub_key[0]
    m = m * phiInv % pub_key[0]
    return m % pub_key[0]

def decrypt_vertices(vertices, paillier_keys) -> np.array:
    print("Decryption")
    
    for i_v, _ in enumerate(vertices):
        for i_c, _ in enumerate(vertices[i_v]):
            vertices[i_v][i_c] = decrypt(vertices[i_v][i_c], paillier_keys["secret"], paillier_keys["public"])

    return vertices

def encrypt_vertices(vertices, public_key) -> np.array:    
    print("Encryption")

    for i_v, _ in enumerate(vertices):
        for i_c, _ in enumerate(vertices[i_v]):
            vertices[i_v][i_c] = encrypt(vertices[i_v][i_c], public_key)

    return vertices
