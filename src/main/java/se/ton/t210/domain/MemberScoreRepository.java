package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import java.time.LocalDate;
import java.util.List;
import java.util.Set;

public interface MemberScoreRepository extends JpaRepository<MemberScore, Long> {

    List<MemberScore> findAllByMemberIdAndCreatedAt(Long memberId, LocalDate month);

    List<MemberScore> findAllByMemberIdInAndCreatedAtOrderByScore(Set<Long> memberIds, LocalDate month);

}
