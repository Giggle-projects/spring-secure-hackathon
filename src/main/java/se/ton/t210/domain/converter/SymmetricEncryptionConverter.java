package se.ton.t210.domain.converter;

import javax.persistence.AttributeConverter;
import javax.persistence.Converter;
import se.ton.t210.utils.encript.SeedUtils;

@Converter
public class SymmetricEncryptionConverter implements AttributeConverter<String, String> {

  private static final String KEY = "sdafjksaldfjklsadjfkljkjslkafjlsdaf";

  @Override
  public String convertToDatabaseColumn(String ccNumber) {
      return SeedUtils.encode(KEY, ccNumber);
  }

  @Override
  public String convertToEntityAttribute(String dbData) {
      return SeedUtils.decode(KEY, dbData);
  }
}
