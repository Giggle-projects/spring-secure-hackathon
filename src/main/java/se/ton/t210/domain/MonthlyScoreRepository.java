package se.ton.t210.domain;

import org.springframework.data.jpa.repository.JpaRepository;

import javax.persistence.Entity;

public interface MonthlyScoreRepository extends JpaRepository<MonthlyScore, Long> {

}
