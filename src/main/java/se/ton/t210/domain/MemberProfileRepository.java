package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface MemberProfileRepository extends JpaRepository<MemberProfileImage, Long> {
}
