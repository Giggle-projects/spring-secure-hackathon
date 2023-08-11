package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class TakenScoreResponse {

    private final int score;

    public TakenScoreResponse(int score) {
        this.score = score;
    }

    public static TakenScoreResponse of(int takenScore) {
        return new TakenScoreResponse(takenScore);
    }
}
