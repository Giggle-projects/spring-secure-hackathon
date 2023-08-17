package se.ton.t210.utils.data.initData;

import se.ton.t210.domain.type.ApplicationType;

import java.util.Random;

// ML
// id,member_id,applicationType,score,month
public class MLTrainingCsvLine implements CsvLineStrategy {

    private static final Random RANDOM = new Random();
    private Long id = 0L;
    private Long userId = 0L;
    private int month = 0;
    private String applicationTypeName;

    @Override
    public String retrieveNewLine() {
        setColumnElement();
        if (id < 7000) {
            return String.join(",",
                    String.valueOf(id),
                    String.valueOf(userId / 12),
                    applicationTypeName,
                    randomNum(40, 50),
                    String.valueOf(month)
            );
        } else {
            return String.join(",",
                    String.valueOf(id),
                    String.valueOf(userId / 12),
                    applicationTypeName,
                    randomNum(5, 49),
                    String.valueOf(month)
            );
        }
    }

    private void setColumnElement() {
        id++;
        userId++;
        month = month % 12;
        if (month == 0) {
            applicationTypeName = randomApplicationTypeStandardName();
        }
        month++;
    }

    private String randomNum(int min, int max) {
        return String.valueOf(RANDOM.nextInt(max - min + 1) + min);
    }

    private String randomApplicationTypeStandardName() {
        ApplicationType[] applicationTypes = ApplicationType.values();
        int randomNum = RANDOM.nextInt(applicationTypes.length);
        return applicationTypes[randomNum].getStandardName();
    }
}
