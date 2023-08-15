package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;

@Getter
public class LoginMemberInfo {

    private final Long id;
    private final String name;
    private final String email;
    private final ApplicationType applicationType;

    public LoginMemberInfo(Long id, String name, String email, ApplicationType applicationType) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.applicationType = applicationType;
    }

    public static LoginMemberInfo of(Member member) {
        return new LoginMemberInfo(
            member.getId(),
            member.getName(),
            member.getEmail(),
            member.getApplicationType()
        );
    }
}
