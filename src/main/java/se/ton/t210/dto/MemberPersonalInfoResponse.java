package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class MemberPersonalInfoResponse {

    private final ApplicationType applicationType;
    private final String name;
    private final String email;

    public MemberPersonalInfoResponse(ApplicationType applicationType, String name, String email) {
        this.applicationType = applicationType;
        this.name = name;
        this.email = email;
    }
}
