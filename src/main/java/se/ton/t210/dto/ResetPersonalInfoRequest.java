package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class ResetPersonalInfoRequest {

    private final ApplicationType applicationType;
    private final String password;

    public ResetPersonalInfoRequest(ApplicationType applicationType, String password) {
        this.applicationType = applicationType;
        this.password = password;
    }
}
