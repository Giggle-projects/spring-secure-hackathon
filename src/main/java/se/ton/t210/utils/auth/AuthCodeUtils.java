package se.ton.t210.utils.auth;

import java.security.SecureRandom;

public class AuthCodeUtils {

    public static String generate(int length) {
        SecureRandom secureRandom = new SecureRandom();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length; i++) {
            builder.append(secureRandom.nextInt(10));
        }
        return builder.toString();
    }
}
