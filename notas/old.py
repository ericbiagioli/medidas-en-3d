
class LastNSeconds:
  def __init__(self):
    self.captured = []
    self.TIME_INTERVAL = 3.0

  def add(self, p):
    t = datetime.now()
    self.captured.append( (t, p))
    i = 0
    while i < len(self.captured) and (t - self.captured[i][0]).total_seconds() > self.TIME_INTERVAL:
      i += 1
    self.captured = self.captured[i:]


  def ncaptures(self):
    return len(self.captured)

  def stats(self):
    points = np.array([p for (_, p) in self.captured])
    mean = points.mean(axis = 0)
    std = points.std(axis = 0)
    return mean, std

  def ok(self):
    points = np.array([p for _, p in self.captured])
    mean = points.mean(axis = 0)
    std = points.std(axis = 0)
    ok1 = (len(self.captured) > 9)
    ok2 = (std < 0.001).all()
    return (ok1 and ok2), mean, std
