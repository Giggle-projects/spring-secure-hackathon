package se.ton.t210.domain;

import se.ton.t210.domain.type.ApplicationType;

import javax.persistence.metamodel.SingularAttribute;
import javax.persistence.metamodel.StaticMetamodel;

@StaticMetamodel(MonthlyScore.class)
public class MonthlyScore_ {
    public static volatile SingularAttribute<MonthlyScore, Long> id;
    public static volatile SingularAttribute<MonthlyScore, ApplicationType> applicationType;
    public static volatile SingularAttribute<MonthlyScore, Integer> score;
    public static volatile SingularAttribute<MonthlyScore, Long> memberId;
}
