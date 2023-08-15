package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ScoreResponse {

    private final int score;
    private final int maxScore;

    public ScoreResponse(int score, int maxScore) {
        this.score = score;
        this.maxScore = maxScore;
    }

    public ScoreResponse(int score) {
        this(score, 0);
    }
}
