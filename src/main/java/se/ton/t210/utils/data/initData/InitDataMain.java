package se.ton.t210.utils.data.initData;

import java.io.IOException;

public class InitDataMain {

    public static void main(String[] args) throws IOException {
        var csvLineStrategy = new MLTrainingCsvLine();
        var initDataCSV = new InitDataCSV(csvLineStrategy, "ml_training_000_001.csv", 100_000);
        initDataCSV.setInitialLine("id,member_id,applicationType,score,month");
        initDataCSV.generate();
    }
}
