package se.ton.t210.utils.encript;

import lombok.Getter;

@Getter
public class EncryptedCode {

   private final String code;
   private final String saltKey;

  public EncryptedCode(String code, String saltKey) {
    this.code = code;
    this.saltKey = saltKey;
  }
}
