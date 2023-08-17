package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

import java.util.Optional;

@Getter
public class ResetPersonalInfoRequest {

    private final Optional<ApplicationType> applicationType;

    private final Optional<String> password;

    public ResetPersonalInfoRequest(Optional<ApplicationType> applicationType, Optional<String> password) {
        this.applicationType = applicationType;
        this.password = password;
    }
}
