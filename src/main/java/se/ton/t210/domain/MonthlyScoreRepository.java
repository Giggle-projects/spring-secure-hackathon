package se.ton.t210.domain;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import se.ton.t210.domain.type.ApplicationType;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface MonthlyScoreRepository extends JpaRepository<MonthlyScore, Long>, JpaSpecificationExecutor<MonthlyScore> {

    int countByApplicationType(ApplicationType applicationType);

    int countByApplicationTypeAndYearMonth(ApplicationType applicationType, LocalDate yearMonth);

    Optional<MonthlyScore> findByMemberIdAndYearMonth(Long memberId, LocalDate yearMonth);

    List<MonthlyScore> findAllByApplicationTypeAndYearMonth(ApplicationType applicationType, LocalDate yearMonth, PageRequest page);

    void deleteAllByMemberIdAndYearMonth(Long memberId, LocalDate yearMonth);

    List<MonthlyScore> findAllByMemberId(Long memberId);

    int countByApplicationTypeAndYearMonthAndScoreGreaterThan(ApplicationType applicationType, LocalDate yearMonth, int score);
}
