import unittest
from src.dashboard.alarms import PPETracker,associate
from src.dashboard.server import number

def detection(cls,box):return {'class_id':cls,'confidence':.9,'box':box}

class PPETests(unittest.TestCase):
    def test_equipment_is_assigned_to_one_person(self):
        ds=[detection(0,[10,10,150,300]),detection(0,[80,10,210,300]),detection(1,[90,20,115,45])]
        people=associate(ds,400,400)
        self.assertEqual(sum('helmet' in p['present'] for p in people),1)
    def test_equipment_outside_person_zone_is_not_assigned(self):
        people=associate([detection(0,[10,10,150,300]),detection(1,[50,250,80,280])],400,400)
        self.assertIn('helmet',people[0]['missing'])
    def test_persistence_requires_duration_and_observations(self):
        people=associate([detection(0,[10,10,150,300])],400,400);t=PPETracker()
        self.assertFalse(t.update(people,0)[0]['alert'])
        self.assertFalse(t.update(people,1)[0]['alert'])
        self.assertTrue(t.update(people,2.1)[0]['alert'])
    def test_missed_frame_resets_persistence(self):
        p=associate([detection(0,[10,10,150,300])],400,400);t=PPETracker()
        t.update(p,0);t.update(p,1);t.update([],1.5)
        self.assertFalse(t.update(p,2.1)[0]['alert'])
    def test_ppe_reappearance_clears_alert(self):
        p=associate([detection(0,[10,10,150,300])],400,400);t=PPETracker()
        for now in [0,1,2.1]:t.update(p,now)
        clean=associate([detection(0,[10,10,150,300]),detection(1,[40,20,70,40]),detection(2,[35,80,110,170])],400,400)
        result=t.update(clean,3)[0]
        self.assertFalse(result['alert']);self.assertEqual(result['assessment'],'PPE detected')
    def test_clipped_person_never_raises_missing_alert(self):
        p=associate([detection(0,[0,10,150,300])],400,400);t=PPETracker()
        for now in [0,1,2,3]:result=t.update(p,now)[0]
        self.assertTrue(result['uncertain']);self.assertFalse(result['alert'])
    def test_requirement_changes_reset_streak(self):
        ds=[detection(0,[10,10,150,300])];t=PPETracker()
        t.update(associate(ds,400,400,['helmet']),0);t.update(associate(ds,400,400,['helmet']),1)
        self.assertFalse(t.update(associate(ds,400,400,['vest']),2.1)[0]['alert'])
    def test_invalid_numeric_thresholds_rejected(self):
        for value in [float('nan'),float('inf'),-1,2]:
            with self.assertRaises(ValueError):number(value,0,1)

if __name__=='__main__':unittest.main()
