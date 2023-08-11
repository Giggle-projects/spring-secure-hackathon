package se.ton.t210.domain.converter;

import java.time.LocalDate;
import javax.persistence.AttributeConverter;
import javax.persistence.Converter;

@Converter
public class ScoreRecordYearAndMonthConverter implements AttributeConverter<LocalDate, String> {

    private static final String DELIMITER = ":";

    @Override
    public String convertToDatabaseColumn(LocalDate attribute) {
        int year = attribute.getYear();
        int month = attribute.getMonth().getValue();
        return year + DELIMITER + month;
    }

    @Override
    public LocalDate convertToEntityAttribute(String dbData) {
        int year = Integer.parseInt(dbData.split(DELIMITER)[0]);
        int month = Integer.parseInt(dbData.split(DELIMITER)[1]);
        return LocalDate.of(year, month, 1);
    }
}
