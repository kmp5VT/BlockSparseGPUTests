struct Model{T} end
struct OneDHeis end

struct TwoDHubbSmall end
struct TwoDHubbMed end
struct TwoDHubbLarge end

model_type(::Model{T}) where {T} = T
