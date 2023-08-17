package se.ton.t210.domain;

import lombok.Getter;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;

@Getter
@Entity
public class MemberProfileImage {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    private Long memberId;

    private String imageUrl;

    public MemberProfileImage() {
    }

    public MemberProfileImage(Long id, Long memberId, String imageUrl) {
        this.id = id;
        this.memberId = memberId;
        this.imageUrl = imageUrl;
    }

    public MemberProfileImage(Long memberId, String imageUrl) {
        this.id = null;
        this.memberId = memberId;
        this.imageUrl = imageUrl;
    }
}
