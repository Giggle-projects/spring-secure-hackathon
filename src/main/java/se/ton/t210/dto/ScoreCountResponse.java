package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class ScoreCountResponse {

    private final int count;

    public ScoreCountResponse(int count) {
        this.count = count;
    }
}
