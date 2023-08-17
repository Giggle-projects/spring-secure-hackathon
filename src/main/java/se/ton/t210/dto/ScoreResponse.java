package se.ton.t210.dto;

import lombok.Getter;

import javax.validation.constraints.NotBlank;

@Getter
public class ScoreResponse {

    @NotBlank
    private final int score;

    public ScoreResponse(int score) {
        this.score = score;
    }
}
