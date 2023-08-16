package se.ton.t210.utils.auth;

import java.security.SecureRandom;
import java.util.Random;

public class RandomCodeUtils {

    private static final Random RANDOM = new SecureRandom();

    public static String generate() {
        byte[] salt = new byte[16];
        RANDOM.nextBytes(salt);
        final StringBuilder sb = new StringBuilder();
        for(byte b : salt) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
