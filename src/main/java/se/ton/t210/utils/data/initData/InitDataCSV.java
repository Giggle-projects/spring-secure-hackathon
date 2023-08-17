package se.ton.t210.utils.data.initData;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class InitDataCSV {

    private static final int UNIT_COUNT = 100_000;

    private final CsvLineStrategy csvLineStrategy;
    private final String fileName;
    private final int dataCount;
    private String initialLine;

    public InitDataCSV(CsvLineStrategy csvLineStrategy, String fileName, int dataCount) {
        this.csvLineStrategy = csvLineStrategy;
        this.fileName = fileName;
        this.dataCount = dataCount;
    }

    public void generate() throws IOException {
        generate(false);
    }

    public void generate(boolean appendEnable) throws IOException {
        try (
                final var bw = new BufferedWriter(new FileWriter(fileName, appendEnable))
        ) {
            if (initialLine != null) {
                bw.append(initialLine);
            }
            for (int i = 0; i < dataCount / UNIT_COUNT; i++) {
                for (int j = 0; j < UNIT_COUNT; j++) {
                    bw.append("\n").append(csvLineStrategy.retrieveNewLine());
                }
            }
            for (int i = 0; i < dataCount % UNIT_COUNT; i++) {
                bw.append("\n").append(csvLineStrategy.retrieveNewLine());
            }
        }
    }

    public void setInitialLine(String line) {
        if (line != null) {
            this.initialLine = line;
        }
    }
}
