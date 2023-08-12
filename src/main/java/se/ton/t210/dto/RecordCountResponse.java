package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class RecordCountResponse {

    private final ApplicationType applicationType;
    private final int count;

    public RecordCountResponse(ApplicationType applicationType, int count) {
        this.applicationType = applicationType;
        this.count = count;
    }
}
