package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface MemberProfileRepository extends JpaRepository<MemberProfileImage, Long> {

    Optional<MemberProfileImage> findByMemberId(Long memberId);
}
