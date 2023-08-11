package se.ton.t210.domain;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;

public interface JudgingItemRepository extends JpaRepository<JudgingItem, Long> {

    List<JudgingItem> findAllByApplicationType(ApplicationType applicationType);

}
