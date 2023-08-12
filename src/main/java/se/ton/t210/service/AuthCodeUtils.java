package se.ton.t210.service;

import java.security.SecureRandom;

public class AuthCodeUtils {

    public static String createAuthNumberCode(int length) {
        try {
            SecureRandom secureRandom = new SecureRandom();
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++) {
                builder.append(secureRandom.nextInt(10));
            }
            return builder.toString();
        } catch (Exception e) {
            throw new IllegalArgumentException("email authentication code create error");
        }
    }
}
