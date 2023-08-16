package se.ton.t210.utils.encript;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import se.ton.t210.utils.auth.RandomCodeUtils;

public class EncryptUtils {

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
       return RandomCodeUtils.generate();
    }
}
