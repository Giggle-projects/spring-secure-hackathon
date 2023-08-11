package se.ton.t210.domain;

import javax.persistence.Entity;
import javax.persistence.EnumType;
import javax.persistence.Enumerated;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import lombok.Getter;

@Getter
@Entity
public class JudgingItem {

    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Id
    private Long id;

    @Enumerated(EnumType.STRING)
    private ApplicationType applicationType;

    private String name;

    public JudgingItem() {
    }

    public JudgingItem(Long id, ApplicationType applicationType, String name) {
        this.id = id;
        this.applicationType = applicationType;
        this.name = name;
    }

    public JudgingItem(ApplicationType applicationType, String name) {
        this(null, applicationType, name);
    }
}
