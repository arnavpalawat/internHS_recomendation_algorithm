class Jobs:
  def __init__(self, id, description, title):
      self.id = id
      self.description = description
      self.title = title

  @classmethod
  def from_firebase(cls, firebase_data):
      id = firebase_data.get("id")
      description = firebase_data.get("description")
      title = firebase_data.get("title")
      return cls(id, description, title)
