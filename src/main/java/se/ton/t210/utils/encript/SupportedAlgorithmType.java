package se.ton.t210.utils.encript;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

public enum SupportedAlgorithmType {
  SHA256("SHA-256"),
  SHA3_256("SHA3-256"),
  KECCAK_256("Keccak-256");

  private final String instanceName;

  SupportedAlgorithmType(String instanceName) {
    this.instanceName = instanceName;
  }

  public MessageDigest messageDigestInstance() throws NoSuchAlgorithmException {
    return MessageDigest.getInstance(this.instanceName);
  }
}
