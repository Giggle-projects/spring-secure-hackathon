package se.ton.t210.utils.encript;

import static java.nio.charset.StandardCharsets.UTF_8;

import java.util.Arrays;
import java.util.Base64;
import se.ton.t210.exception.InnerServiceException;

public class SeedUtils {

  public static String encode(String encodedSeedKey, String plainText) {
    final byte[] seedKey = Base64.getDecoder().decode(encodedSeedKey);
    final byte[] userKey = Arrays.copyOfRange(seedKey, 0, 16);
    final byte[] IV = Arrays.copyOfRange(seedKey, 16, 32);

    byte[] encryptedInfo = KISA_SEED_CBC.SEED_CBC_Encrypt(
        userKey, IV,
        plainText.getBytes(UTF_8), 0, plainText.getBytes(UTF_8).length
    );

    byte[] encodedEncryptedInfo = Base64.getEncoder().encode(encryptedInfo);
    return new String(encodedEncryptedInfo, UTF_8);
  }

  public static String decode(String encodedSeedKey, String encodedText) {
    try {
      byte[] seedKey = Base64.getDecoder().decode(encodedSeedKey);
      final byte[] userKey = Arrays.copyOfRange(seedKey, 0, 16);
      final byte[] IV = Arrays.copyOfRange(seedKey, 16, 32);

      byte[] cipherInfo = Base64.getDecoder().decode(encodedText);
      byte[] decryptedInfo = KISA_SEED_CBC.SEED_CBC_Decrypt(
          userKey, IV,
          cipherInfo, 0, cipherInfo.length
      );

      return new String(decryptedInfo, UTF_8);
    } catch (Exception e) {
      throw new InnerServiceException("fail seed cbc decrypt.");
    }
  }

}
