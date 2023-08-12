package se.ton.t210.dto;

import lombok.Getter;

@Getter
public class MemberScoreResponse {

    public final int score;

    public MemberScoreResponse(int score) {
        this.score = score;
    }
}
