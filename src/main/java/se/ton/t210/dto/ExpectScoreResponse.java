package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ExpectScoreResponse {

    private final int currentScore;
    private final int currentPercentile;
    private final int expectedPassPercent;

    public ExpectScoreResponse(int currentScore, int currentPercentile, int expectedPassPercent) {
        this.currentScore = currentScore;
        this.currentPercentile = currentPercentile;
        this.expectedPassPercent = expectedPassPercent;
    }
}
