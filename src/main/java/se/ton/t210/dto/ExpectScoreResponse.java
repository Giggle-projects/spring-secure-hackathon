package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ExpectScoreResponse {

    private final int currentScore;
    private final int expectedScore;
    private final int expectedGrade;

    public ExpectScoreResponse(int currentScore, int expectedScore, int expectedGrade) {
        this.currentScore = currentScore;
        this.expectedScore = expectedScore;
        this.expectedGrade = expectedGrade;
    }
}
