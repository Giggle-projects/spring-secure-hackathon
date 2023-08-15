package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ExpectScoreResponse {

    private final int currentScore;
    private final float currentPercentile;
    private final float expectedPassPercent;

    public ExpectScoreResponse(int currentScore, float currentPercentile, float expectedPassPercent) {
        this.currentScore = currentScore;
        this.currentPercentile = currentPercentile;
        this.expectedPassPercent = expectedPassPercent;
    }
}
