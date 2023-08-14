package se.ton.t210.domain;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.jpa.repository.JpaRepository;
import se.ton.t210.domain.type.ApplicationType;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

public interface MonthlyScoreRepository extends JpaRepository<MonthlyScore, Long> {

    int countByApplicationType(ApplicationType applicationType);

    Optional<MonthlyScore> findByMemberIdAndYearMonth(Long memberId, LocalDate yearMonth);

    List<MonthlyScore> findAllByApplicationTypeAndYearMonth(ApplicationType applicationType, LocalDate yearMonth, PageRequest page);

    void deleteAllByMemberIdAndYearMonth(Long memberId, LocalDate yearMonth);
}
