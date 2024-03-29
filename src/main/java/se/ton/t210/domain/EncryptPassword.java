package se.ton.t210.domain;

import java.security.NoSuchAlgorithmException;
import lombok.Getter;
import se.ton.t210.exception.InnerServiceException;
import se.ton.t210.utils.encript.SHA256Utils;

@Getter
public class EncryptPassword {

  private final String encrypted;
  private final String salt;

  private EncryptPassword(String encrypted, String salt) {
    this.encrypted = encrypted;
    this.salt = salt;
  }

  public static EncryptPassword encryptFrom(String plainText) {
    try {
      final String salt = SHA256Utils.generateSalt();
      final String encryptedNewPassword = SHA256Utils.encrypt(plainText, salt);
      return new EncryptPassword(encryptedNewPassword, salt);
    } catch (NoSuchAlgorithmException e) {
      throw new InnerServiceException("Inner service error in encrypt");
    }
  }

  public static EncryptPassword encryptFrom(String plainText, String salt) {
    try {
      final String encryptedNewPassword = SHA256Utils.encrypt(plainText, salt);
      return new EncryptPassword(encryptedNewPassword, salt);
    } catch (NoSuchAlgorithmException e) {
      throw new InnerServiceException("Inner service error in encrypt");
    }
  }
}
