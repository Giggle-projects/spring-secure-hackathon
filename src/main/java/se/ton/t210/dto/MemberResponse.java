package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.Member;
import se.ton.t210.domain.type.ApplicationType;
import se.ton.t210.domain.type.Gender;

@Getter
public class MemberResponse {

    private final Long id;
    private final String name;
    private final String email;
    private final ApplicationType applicationType;
    private final String applicationTypeName;
    private final Gender gender;

    public MemberResponse(Long id, String name, String email, ApplicationType applicationType, String applicationTypeName, Gender gender) {
        this.id = id;
        this.name = name;
        this.email = email;
        this.applicationType = applicationType;
        this.applicationTypeName = applicationTypeName;
        this.gender = gender;
    }

    public static MemberResponse of(Member member) {
        return new MemberResponse(
            member.getId(),
            member.getName(),
            member.getEmail(),
            member.getApplicationType(),
            member.getApplicationType().getName(),
            member.getGender()
        );
    }
}
