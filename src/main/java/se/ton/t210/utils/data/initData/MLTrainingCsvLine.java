package se.ton.t210.utils.data.initData;

import se.ton.t210.domain.type.ApplicationType;

import java.util.Random;

// ML
// id,applicationType,score,month
public class MLTrainingCsvLine implements CsvLineStrategy {

    private static final Random RANDOM = new Random();
    private Long id = 0L;

    @Override
    public String retrieveNewLine() {
        id++;
        return String.join(",",
                String.valueOf(id),
                randomApplicationTypeStandardName(),
                randomNum(5, 50),
                randomNum(1, 12)
        );
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
