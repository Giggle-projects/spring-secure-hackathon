package se.ton.t210.domain;

import java.time.LocalDate;
import java.util.List;
import java.util.Set;
import org.springframework.data.jpa.repository.JpaRepository;
import se.ton.t210.domain.type.ApplicationType;

public interface MemberScoreRepository extends JpaRepository<MemberScore, Long> {

    List<MemberScore> findAllByMemberIdAndCreatedAt(Long memberId, LocalDate month);

    List<MemberScore> findAllByMemberIdInAndCreatedAtOrderByScore(Set<Long> memberIds, LocalDate month);

}
