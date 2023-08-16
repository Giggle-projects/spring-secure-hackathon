package se.ton.t210.utils.encript;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.util.Random;

public class EncryptUtils {

    private static final Random RANDOM = new SecureRandom();

    public static String encrypt(SupportedAlgorithmType algorithm, String planText, String salt) throws NoSuchAlgorithmException {
        final MessageDigest md = algorithm.messageDigestInstance();
        md.update(salt.getBytes());
        final byte[] digest = md.digest(planText.getBytes(StandardCharsets.UTF_8));
        return bytesToHex(digest);
    }

    private static String bytesToHex(byte[] hash) {
        final StringBuilder hexString = new StringBuilder(2 * hash.length);
        for (byte b : hash) {
          String hex = Integer.toHexString(0xff & b);
          if (hex.length() == 1) {
            hexString.append('0');
          }
          hexString.append(hex);
        }
        return hexString.toString();
    }

    public static String generateSalt() {
      byte[] salt = new byte[16];
      RANDOM.nextBytes(salt);
      final StringBuilder sb = new StringBuilder();
      for(byte b : salt) {
        sb.append(String.format("%02x", b));
      }
      return sb.toString();
    }
}
