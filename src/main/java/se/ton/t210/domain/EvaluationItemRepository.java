package se.ton.t210.domain;

import java.util.List;
import org.springframework.data.jpa.repository.JpaRepository;
import se.ton.t210.domain.type.ApplicationType;

public interface EvaluationItemRepository extends JpaRepository<EvaluationItem, Long> {

    List<EvaluationItem> findAllByApplicationType(ApplicationType applicationType);

}
