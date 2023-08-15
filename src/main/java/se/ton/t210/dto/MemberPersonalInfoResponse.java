package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class MemberPersonalInfoResponse {

    private final String name;
    private final String email;
    private final ApplicationType applicationType;

    public MemberPersonalInfoResponse(String name, String email, ApplicationType applicationType) {
        this.name = name;
        this.email = email;
        this.applicationType = applicationType;
    }
}
