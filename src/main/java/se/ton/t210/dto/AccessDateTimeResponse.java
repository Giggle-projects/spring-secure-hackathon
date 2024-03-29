package se.ton.t210.dto;

import lombok.Getter;
import se.ton.t210.domain.AccessDateTime;
import se.ton.t210.domain.Member;

import java.time.LocalDateTime;

@Getter
public class AccessDateTimeResponse {

    private final LocalDateTime dateTime;
    private final Long memberId;
    private final String memberName;
    private final String memberEmail;
    private final String memberEncryptedPassword;

    public AccessDateTimeResponse(LocalDateTime dateTime, Long memberId, String memberName, String memberEmail, String memberEncryptedPassword) {
        this.dateTime = dateTime;
        this.memberId = memberId;
        this.memberName = memberName;
        this.memberEmail = memberEmail;
        this.memberEncryptedPassword = memberEncryptedPassword;
    }

    public static AccessDateTimeResponse of(AccessDateTime it, Member member) {
        return new AccessDateTimeResponse(
            it.getAccessTime(),
            member.getId(),
            member.getName(),
            member.getEmail(),
            member.getPassword()
        );
    }
}
