package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class ScoreResponse {

    private final int score;

    public ScoreResponse(int score) {
        this.score = score;
    }
}
