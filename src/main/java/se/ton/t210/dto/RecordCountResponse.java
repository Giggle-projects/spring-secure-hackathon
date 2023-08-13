package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class RecordCountResponse {

    private final int count;

    public RecordCountResponse(int count) {
        this.count = count;
    }
}
