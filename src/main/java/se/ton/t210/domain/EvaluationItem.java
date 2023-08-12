package se.ton.t210.domain;

import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;
import se.ton.t210.domain.type.ApplicationType;

@Getter
@Entity
public class EvaluationItem {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    @Enumerated(EnumType.STRING)
    private ApplicationType applicationType;

    private String name;

    public EvaluationItem() {
    }

    public EvaluationItem(Long id, ApplicationType applicationType, String name) {
        this.id = id;
        this.applicationType = applicationType;
        this.name = name;
    }

    public EvaluationItem(ApplicationType applicationType, String name) {
        this(null, applicationType, name);
    }
}
